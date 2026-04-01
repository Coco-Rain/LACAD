import cadquery as cq

result = (
cq.Workplane()
.center(10, 20)
.tag("centered")
.hLine(5)
.polarLine(4, 135)
.hLineTo(0)
.mirrorY()
.extrude(2)
.workplaneFromTagged("centered")
.center(0, 2)
.rect(11, 2, centered=True)
.extrude(2)
.edges("|Z")
.chamfer(0.5)
)
cq.exporters.export(result, 'GT.stl')
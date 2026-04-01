import cadquery as cq

result = (
cq.Workplane("XY")
.box(55, 8, 5.5)
.translate((22.5, 0, 2.5))
.faces(">Z").edges().fillet(3.99)
.edges("not %LINE").fillet(3)
.faces(">Z").workplane()
.pushPoints([(15, 0)])
.cboreHole(3.3, 5.8, 4)
)
cq.exporters.export(result, 'GT.stl')
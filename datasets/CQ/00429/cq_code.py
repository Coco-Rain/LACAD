import cadquery as cq

result = (
cq.Workplane('XZ', origin=(0, 0, 0))
.moveTo(2, 4)
.vLineTo(0)
.tangentArcPoint((-2, 0), relative=False)
.vLineTo(4)
.close()
.extrude(6)
)
cq.exporters.export(result, 'GT.stl')
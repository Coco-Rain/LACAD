import cadquery as cq

result = (
cq.Workplane('XZ', origin=(0, 0, 0))
.moveTo(2.34, 3.9)
.vLineTo(0)
.tangentArcPoint((-2.34, 0), relative=False)
.vLineTo(3.9)
.close()
.extrude(5.88)
.rotate((0, 0, 0), (0, 0, 1), 180)
)
cq.exporters.export(result, 'GT.stl')